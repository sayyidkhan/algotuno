import * as tf from "@tensorflow/tfjs-node";
import cloneDeep from 'clone-deep';

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
    for (let i = 0; i < processed_stock_data.length; i++) {
        const curr_counter = i + 1;
        if (curr_counter % 2 === 0) {
            let curr_item = processed_stock_data[i];
            labelled_list.push(curr_item);
        }
    }
    return labelled_list;
}

function convert_training_data_into_list(data) {
    return data.map(e => {
        return e.map(f => {
            return [f.Close, f.Epoch];
        })
    });
}

function convert_label_data_into_list(data) {
    const result_list = [];
    for (let i = 0; i < data.length; i++) {
        const next_mth_price = data[i];
        result_list.push([next_mth_price.Close, next_mth_price.Epoch]);
    }
    return result_list;
}

function prepare_xs_ys_dataset(_stock_data, _year, _window) {
    //data for processing
    let training_data = get_stock_data(_stock_data, _year, _window); // training data is [27, 2]
    let labelled_data = [...training_data.slice(1, training_data.length)];
    // 2. drop the 13th month training data record
    training_data.pop();
    // 3. convert into array containing array's only
    training_data = convert_training_data_into_list(training_data);
    labelled_data = convert_training_data_into_list(labelled_data);
    // 4. remove dates
    let remove_date_training_data = training_data.map(x => x.map(y => y[0]));
    let remove_data_labelled_data = labelled_data.map(x => x.map(y => y[0]));
    // 5. convert arrays into tensor
    const tensor_training_data = tf.tidy(() => tf.tensor2d(remove_date_training_data));
    const tensor_labelled_data = tf.tidy(() => tf.tensor2d(remove_data_labelled_data));

    return {
        'original_xs_data': training_data,
        'original_ys_data': labelled_data,
        'training_xs_array': remove_date_training_data,
        'training_ys_array': remove_data_labelled_data,
        'tensor_xs_data': tensor_training_data,
        'tensor_ys_data': tensor_labelled_data,
    }
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
        tf.layers.dense({
            units: (training_data.size * 3),
            activation: 'relu'
        })
    );
    model.add(
        tf.layers.dense({
            units: (training_data.size * 2),
            activation: 'relu'
        })
    );
    model.add(
        tf.layers.dense({
            units: (training_data.size * 1),
            activation: "elu",
        })
    );
    model.add(tf.layers.dense({
        units: labelled_data.shape[1],
        // here the input shape is "inferred from the previous shape"
        activation: 'linear',
    }));


    const ALPHA = 0.0005;
    model.compile({optimizer: tf.train.adam(ALPHA), loss: "meanSquaredError"});
    return model;
});

async function train(xs, ys, iterations, _model, print_training_log) {
    const loss_output = [];
    const config = {epochs: 1, verbose: print_training_log === true ? 1 : 0};
    for (let i = 0; i < iterations; i++) {
        const response = await _model.fit(xs, ys, config);
        const epochs_output = {"epoch": ((i + 1) * config.epochs), "loss": response.history.loss[0]};
        loss_output.push(epochs_output);
        // display error loss
        if (print_training_log === true) {
            console.log(epochs_output);
        }
    }

    return {
        "loss_output": loss_output
    }
}


async function predict_v(_model, _training_data, _labelled_data) {
    const _prediction = await _model.predict(_training_data);
    console.log("perform prediction");
    // let _loss_output = await train_model.then(loss_output => loss_output);
    // merge the dates back in
    let _predict = _prediction.arraySync();

    function merge_dates_from_original_dataset(_original_dataset, _predicted_data) {
        return _predicted_data.map((element, index) => {
            return [
                [element[0], new Date(_original_dataset[index][0][1])],
                [element[1], new Date(_original_dataset[index][1][1])],
            ]
        });
    }

    _predict = merge_dates_from_original_dataset(_labelled_data, _predict);
    return _predict;
}

export {
    prepare_xs_ys_dataset,
    nn_model,
    train,
    predict_v
};