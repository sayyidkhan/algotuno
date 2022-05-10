const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const BUYING_INDICATORS_CLASSES = ['BUY', 'SELL', 'HOLD'];
const BUYING_INDICATORS_CLASSES_NO = BUYING_INDICATORS_CLASSES.length;

let rawdata = fs.readFileSync('appl_price.json');
let stock_data = JSON.parse(rawdata);

const clone = (items) => items.map(item => Array.isArray(item) ? clone(item) : item);

const predict_up_down_hold = (curr_price, future_price) => {
    // if price 116.52 and 116. 32 (small price movement) - HOLD
    if (parseInt(curr_price) === parseInt(future_price)) {
        return 0;
    }
    // if curr price greater than buying price - DOWN
    else if (parseInt(curr_price) > parseInt(future_price)) {
        return -1;
    }
    // if curr price lesser than buying price - UP
    else {
        return 1;
    }
};

const covert_to_predict_str = (predict_no) => {
    if (predict_no === -1) {
        return "BUY";
    } else if (predict_no === 0) {
        return "HOLD";
    } else {
        return "SELL";
    }
};

const extract_data = (_data) => {
    return _data.map(obj => {
        return {
            'Date': obj.Date,
            'Epoch': new Date(obj.Date).getTime(),
            'Close': parseFloat(parseFloat(obj.Close).toFixed(2)),
            'Predict': undefined,
            'Next': undefined,
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

const get_next_wk_record_in_curr_record = (_data) => {
    for (let month = 0; month < _data.length - 1; month++) {
        const month_record = _data[month];
        const next_month_record = _data[month + 1];

        const next_month_first_record = next_month_record[0];
        for (let week = 0; week < month_record.length; week++) {
            const curr_week = month_record[week];

            // at the last record, take the next month record
            if (week === month_record.length - 1) {
                const _predict = predict_up_down_hold(curr_week.Close, next_month_first_record.Close);
                curr_week['Next'] = next_month_first_record.Close;
                curr_week['Predict'] = _predict;
            } else {
                const next_week = month_record[week + 1];
                const _predict = predict_up_down_hold(curr_week.Close, next_week.Close);
                curr_week['Next'] = next_week.Close;
                curr_week['Predict'] = _predict;
            }
        }
    }
    return _data;
};


function get_stock_data(year) {
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


    group_mth_stock_data_curr.push(group_mth_stock_data_next[0]); // later aft processing will drop the 13th month

    get_next_wk_record_in_curr_record(group_mth_stock_data_curr);
    group_mth_stock_data_curr.pop(); // remove 13 mth record
    return group_mth_stock_data_curr;
}

function prepare_stock_data(_data) {
    // data is a list of list - so need to loop through it twice
    const xTraining = [];
    const yTraining = [];

    _data.forEach(month_list => {
        const xMonthList = [];
        const yMonthList = [];
        month_list.forEach(week => {
            const weeklyrecord_x = [week.Epoch, week.Close];
            const weeklyrecord_y = [week.Predict, week.Next];
            xMonthList.push(weeklyrecord_x);
            yMonthList.push(weeklyrecord_y);
        });
        // add the month list to the training list
        xTraining.push(xMonthList);
        yTraining.push(yMonthList);
    });

    return {
        'x': xTraining.flat(),
        'y': yTraining.flat(),
    };
}

function get_training_data(year) {
    const stock_data = get_stock_data(year);
    const p_data = prepare_stock_data(stock_data);
    return p_data;
}

function convert_to_tensors(data) {
    const xTraining = data.x;
    const yTraining = data.y;

    const no_of_nodes_x = xTraining.length;
    const node_size_x = xTraining[0].length;
    const xs = tf.tensor2d(xTraining); // training data

    const no_of_nodes_y = yTraining.length;
    const node_size_y = yTraining[0].length;
    const ys = tf.tensor2d(yTraining); // labels
    return {
        'x': xs,
        'y': ys
    };
}

const nn_model = (data) => tf.tidy(() => {
    const model = tf.sequential();

    const data_x_shape = data.x.shape;
    const data_y_shape = data.y.shape;

    // input layer
    model.add(
        tf.layers.dense({
            units: data_x_shape[0], // number of nodes
            inputShape: [data_x_shape[1]], // input shape
            activation: "sigmoid",
        })
    );

    model.add(
        tf.layers.dense({
            units: data_x_shape[0] * 2, // number of nodes
            // here the input shape is "inferred from the previous shape"
            activation: "relu",
        })
    );

    const output = tf.layers.dense({
        units: data_y_shape[1], // output shape
        // here the input shape is "inferred from the previous shape"
        activation: 'linear',
    });
    model.add(output);

    const ALPHA = 0.0005;
    model.compile({optimizer: tf.train.adam(ALPHA), loss: "meanSquaredError"});
    return model;
});


async function train(
    _model,
    _data,
    total_runs,
    epochs) {
    const xTraining = _data.x;
    const yLabel = _data.y;

    for (let i = 0; i < total_runs; i++) {
        const config = {
            epochs: epochs,
        };
        const response = await _model.fit(xTraining, yLabel, config);
        console.log(response.history.loss[0]);
    }
    return _model;
}

// 1. prepare data
let p_data2016 = get_training_data(2016);
console.log(p_data2016);
// p_data2016 = {'x': p_data2016.x.slice(0,1), 'y': p_data2016.y.slice(0,1)};
// const tensor_data2016 = convert_to_tensors(p_data2016);
// const training_data2016 = tensor_data2016.x;
// const label_data2016 = tensor_data2016.y;
//
// // // 2. prepare model
// const model_2016 = nn_model(tensor_data2016);
// console.log(model_2016.summary());
// //
// // // 3. train model
// const total_runs = 10;
// const epochs = 1000;
// train(model_2016, tensor_data2016, total_runs, epochs)
//     .then((_model) => {
//         console.log("training complete");
//         console.log("");
//         console.log(training_data2016.arraySync());
//         let outputs = _model.predict(label_data2016);
//         outputs.print();
//     });