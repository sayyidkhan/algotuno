import * as tf from '@tensorflow/tfjs-node';
import {get_stock_data_from_fs} from "./stock_util.js";
import {get_training_and_labelled_data_from_stock, nn_model, predict, train, forecast} from "./main_v3.js";
import {get_accuracy} from "./display_chart_v3.js";
import {plot} from "nodeplotlib";
import fs from "fs";

let model_dir = 'file://tf_models/lstm_6/model.json';
let rawdata = fs.readFileSync('./stock_price/app_price_latest.json');
let appl_stock_data = JSON.parse(rawdata);

/*** variables ***/
const _dir_to_save_file = 'file://tf_models/lstm_6';
const stock_data = './stock_price/app_price_latest.json';
const window = 4; // every 2 weeks (each whole no represent every week)
const weeks_to_predict = 4; // how many weeks to predict into the future
const epoch = 1000; // no of times to train
const display_training_output = true; // display logs on training output per epoch
const display_model_summary = false;


/*** program logic ***/
// 1. get training and labelled dataset
let extracted_stock_data = get_stock_data_from_fs(stock_data);
const training_dataset_2018 = get_training_and_labelled_data_from_stock(extracted_stock_data, 2018, window);
const validation_dataset_2019 = get_training_and_labelled_data_from_stock(extracted_stock_data, 2019, window);
console.log(validation_dataset_2019.original_xs_data);
const test_dataset_2020 = get_training_and_labelled_data_from_stock(extracted_stock_data, 2020, window);

// 2. load the neural network model
const sample_model = tf.loadLayersModel(model_dir);

// 3. load model
sample_model
    .then((_model) => {
        try {
            console.log("successfully loaded model");
            console.log(_model.summary());
            return _model;
        } catch (e) {
            console.log("unable to load model");
            console.log(e);
        }
    })
    .then((_model) => {
        predict(
            _model,
            validation_dataset_2019.tensor_xs_data,
            validation_dataset_2019.original_ys_data
        ).then(async (_validation_data_result) => {
            const last_index = _validation_data_result.length - 1;
            const most_recent_price = _validation_data_result[last_index][0];
            const most_recent_date = _validation_data_result[last_index][1];


            console.log(most_recent_price, most_recent_date);
            //
            // console.log(training_dataset_2016.tensor_xs_data);


            // return {
            //     'validation_data': _validation_data_result,
            //     'test_data': undefined
            // }
            return _validation_data_result;
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

                let validation_dataset = validation_dataset_2019.original_ys_data.map(x => [x[0], new Date(x[1])]);
                console.log("display result");
                // result_set
                let _actual_result = generate_plot_line(training_dataset_2018.original_xs_data.flat().map(x => [x[0], new Date(x[1])]), "training");
                let _predict_result = generate_plot_line(training_dataset_2018.original_ys_data.map(x => [x[0], new Date(x[1])]), "predict");
                let _validation_result = generate_plot_line(validation_dataset, "validation");
                let _ml_result = generate_plot_line(_predict, "ml result");


                console.log(_actual_result.x);
                //console.log(validation_dataset_2019.original_xs_data.flat().map(x => [x[0], new Date(x[1])]).slice(-1)[0]);

                // display accuracy
                let _model_accuracy = get_accuracy(validation_dataset, _predict);

                return {
                    'actual': _actual_result,
                    'predict': _predict_result,
                    'validation': _validation_result,
                    'ml_result': _ml_result,
                    'model_accuracy': _model_accuracy,
                };
            })
            .then((_predict_obj) => {
                const actual_result = _predict_obj.actual;
                const predict_result = _predict_obj.predict;
                const ml_result = _predict_obj.ml_result;
                const validation_result = _predict_obj.validation;


                console.log(`model accuracy rate:  ${_predict_obj.model_accuracy.in_percentage}%`);
                console.log(`no of correctness:  ${_predict_obj.model_accuracy.in_number}`);
                plot([actual_result, predict_result]);
                plot([actual_result, validation_result, ml_result]);
            });
    });



