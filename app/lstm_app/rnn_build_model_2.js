import * as tf from '@tensorflow/tfjs-node';
import fs from "fs";
import {plot} from "nodeplotlib";

let rawdata = fs.readFileSync('./stock_price/appl_price.json');
let appl_stock_data = JSON.parse(rawdata);

const extract_data = (_data) => {
    return _data.map(obj => {
        return {
            'Date': obj.Date,
            'Epoch': new Date(obj.Date).getTime(),
            'Close': parseFloat(parseFloat(obj.Close).toFixed(2)),
        };
    });

};

const filter_year = (_data, year) => {
    return _data.map(x => x).filter(obj => {
        return new Date(obj.Date).getUTCFullYear() === year
    });
};

const group_according_to_month = (_data) => {
    // basic group either within 1, 7 , 30 days
    const year_list = [];
    // init 12 arrays in year list, for each and every month
    for (let i = 0; i < 12; i++) {
        year_list.push([]);
    }

    // group the data according to month
    _data.forEach(x => {
        const month = new Date(x.Date).getMonth(); // month is alreadly at zero based index where month 1 is 0 in index
        const month_list = year_list[month];
        month_list.push(x);
    });
    return year_list;
};

function rearrange_record_according_to_window(_stock_data, window) {
    const result_list = [];
    let curr_list = [];
    for (let i = 0; i < _stock_data.length; i++) {
        const stock_obj = _stock_data[i];
        if ((i + 1) % window === 0) {
            curr_list.push(stock_obj);
            result_list.push(curr_list);
            curr_list = [];
        } else {
            curr_list.push(stock_obj);
        }
    }

    return result_list;
}


function get_stock_data(stock_data, year, window) {
// NOTE: this dataset only have the date over a 7 day period - so it can only predict every 7 days

// 1. extract the Date and the close price and create the epoch object
    stock_data = extract_data(stock_data);
// 2. filter the curr year
    let stock_data_curr = filter_year(stock_data, year);
// 3. filter the next year
    let stock_data_next = filter_year(stock_data, year + 1);
// 4. put the next week record in the current obj
    let group_mth_stock_data_curr = group_according_to_month(stock_data_curr);
    let group_mth_stock_data_next = group_according_to_month(stock_data_next);
    group_mth_stock_data_curr.push(group_mth_stock_data_next[0]);
// 5. sort the record according to the window
    let flat_stock_data_curr = group_mth_stock_data_curr.flat();

    let result = rearrange_record_according_to_window(flat_stock_data_curr, window);

    return result;
}

function get_labeled_data(processed_stock_data) {
    const labelled_list = [];
    for (let i = 1; i < processed_stock_data.length; i++) {
        let last_item_obj = processed_stock_data[i][0];
        labelled_list.push(last_item_obj);
    }
    return labelled_list;
}

function convert_label_data_into_list(data) {
    const result_list = [];
    for (let i = 0; i < data.length; i++) {
        const next_mth_price = data[i];
        result_list.push([next_mth_price.Close, next_mth_price.Epoch]);
    }
    return result_list;
}

function convert_training_data_into_list(data) {
    return data.map(e => {
        return e.map(f => {
            return [f.Close, f.Epoch];
        })
    });
}

const nn_model = (training_data, labelled_data) => tf.tidy(() => {
    const model = tf.sequential();
    model.add(
        tf.layers.inputLayer({
            inputShape: [training_data.shape[1]],
            units: training_data.size,
        })
    );
    model.add(
        tf.layers.leakyReLU({
            units: (training_data.size * 3),
            activation: 'relu'
        })
    );
    model.add(
        tf.layers.dense({
            units: (training_data.size * 3),
            activation: "elu",
        })
    );
    model.add(tf.layers.dense({
        units: labelled_data.shape[1],
        // here the input shape is "inferred from the previous shape"
        activation: 'linear',
    }));


    const ALPHA = 0.001;
    model.compile({optimizer: tf.train.adam(ALPHA), loss: "meanSquaredError"});
    return model;
});

// 1. get record by the year
const window = 2; // every 2 weeks
let training_data_2016 = get_stock_data(appl_stock_data, 2016, window); // training data is [13, 4]
let labelled_data_2016 = get_labeled_data(training_data_2016);
// 2. drop the 13th month training dat record
training_data_2016.pop();
// 3. convert into array containing array's only
training_data_2016 = convert_training_data_into_list(training_data_2016);
labelled_data_2016 = convert_label_data_into_list(labelled_data_2016); // labelled data [13, 2]

// 4. extract small for testing
const training_data_2016_without_date = training_data_2016.map(x => x.map(y => y[0]));
const labelled_data_2016_without_date = labelled_data_2016.map(x => [x[0]]);

// 5. convert arrays into tensor
const a = tf.tidy(() => tf.tensor2d(training_data_2016_without_date));
const b = tf.tidy(() => tf.tensor2d(labelled_data_2016_without_date));

// 6. setup the neural network model
const sample_model = nn_model(a, b);

// 7. train the model and output
async function train(xs, ys, iterations) {
    for (let i = 0; i < iterations; i++) {
        const config = {
            epochs: 10,
        };
        const response = await sample_model.fit(xs, ys, config);
        console.log(response.history.loss[0]);
    }
}

train(a, b, 25000)
    .then(() => {
        try {
            sample_model.save('file://tf_models/lstm_2');
            console.log("model saved successfully");
        } catch (e) {
            console.log("Unable to save model");
            console.log(e);
        }
    })
    .then(() => {
        console.log("training complete");
        let outputs = sample_model.predict(a);
        return outputs;
    })
    .then((_outputs) => {
        // merge the dates back in
        let _predict = _outputs.arraySync().map(x => x[0]);
        _predict = _predict.map((element, index) => [element, new Date(labelled_data_2016[index][1])]);
        return _predict;
    })
    .then((_predict) => {
        function convert_to_x_and_y_axis(xs_list) {
            let x = [];
            let y = [];
            for (let i = 0; i < xs_list.length; i++) {
                let curr_item = xs_list[i];

                x.push(curr_item[1]);
                y.push(curr_item[0]);
            }

            return {
                'x-axis': x,
                'y-axis': y,
            };
        }

        function generate_plot_line(price_list, name) {
            let obj = convert_to_x_and_y_axis(price_list, window);
            return {x: obj['x-axis'], y: obj['y-axis'], type: 'plot', name: name};
        }

        console.log("display result");
        let _ml_result = generate_plot_line(_predict, "ml result");
        let _predict_result = generate_plot_line(labelled_data_2016.map(x => [x[0], new Date(x[1])]), "predict");
        let _actual_result = generate_plot_line(training_data_2016.flat().map(x => [x[0], new Date(x[1])]), "actual");
        return {
            'actual': _actual_result,
            'predict': _predict_result,
            'ml_result': _ml_result,
        }
    })
    .then((_predict_obj) => {
        const actual_result = _predict_obj.actual;
        const predict_result = _predict_obj.predict;
        const ml_result = _predict_obj.ml_result;
        plot([actual_result, predict_result]);
        plot([actual_result, ml_result]);
    });




