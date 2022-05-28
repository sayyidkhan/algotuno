import * as tf from '@tensorflow/tfjs-node';
import {get_stock_data_from_fs} from "./stock_util.js";
import {get_training_and_labelled_data_from_stock, nn_model, predict, train, forecast} from "./main.js";
import {plot} from "nodeplotlib";


/*** variables ***/
const _dir_to_save_file = 'file://tf_models/lstm_4';
const stock_data = './stock_price/appl_price.json';
const window = 4; // every 2 days (each whole no represent every week)
const weeks_to_predict = 4; // how many weeks to predict into the future
const epoch = 10; // no of times to train
const display_training_output = true; // display logs on training output per epoch
const display_model_summary = false;


/*** program logic ***/

// 1. get training and labelled dataset
let appl_stock_data = get_stock_data_from_fs(stock_data);
const training_dataset_2016 = get_training_and_labelled_data_from_stock(appl_stock_data, 2016, window);
const validation_dataset_2017 = get_training_and_labelled_data_from_stock(appl_stock_data, 2017, window);
const test_dataset_2018 = get_training_and_labelled_data_from_stock(appl_stock_data, 2018, window);

// 2. setup the neural network model
const sample_model = nn_model(
    training_dataset_2016.tensor_xs_data,
    training_dataset_2016.tensor_ys_data
);

// 3. train the model
train(
    training_dataset_2016.tensor_xs_data,
    training_dataset_2016.tensor_ys_data,
    epoch,
    sample_model,
    display_training_output,
)
    .then((_training_output) => {
        // save model
        if (display_model_summary) {
            console.log(sample_model.summary());
        }
        sample_model.save(_dir_to_save_file);
        console.log("model saved successfully");

        return _training_output.loss_output;
    })
    .finally(() => {
        // 4. perform the prediction
        predict(
            sample_model,
            validation_dataset_2017.tensor_xs_data,
            validation_dataset_2017.original_ys_data
        )
            .then(async (_validation_data_result) => {
                const last_index = _validation_data_result.length - 1;
                const most_recent_price = _validation_data_result[last_index][0];
                const most_recent_date = _validation_data_result[last_index][1];
                // console.log(most_recent_price, most_recent_date);
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

                console.log("display result");
                let _actual_result = generate_plot_line(training_dataset_2016.original_xs_data.flat().map(x => [x[0], new Date(x[1])]), "training");
                let _predict_result = generate_plot_line(training_dataset_2016.original_ys_data.map(x => [x[0], new Date(x[1])]), "predict");
                let _validation_result = generate_plot_line(validation_dataset_2017.original_xs_data.flat().map(x => [x[0], new Date(x[1])]), "validation");
                let _ml_result = generate_plot_line(_predict, "ml result");

                return {
                    'actual': _actual_result,
                    'predict': _predict_result,
                    'validation': _validation_result,
                    'ml_result': _ml_result,
                };
            })
            .then((_predict_obj) => {
                const actual_result = _predict_obj.actual;
                const predict_result = _predict_obj.predict;
                const ml_result = _predict_obj.ml_result;
                const validation_result = _predict_obj.validation;

                plot([actual_result, predict_result]);
                plot([actual_result, validation_result, ml_result]);
            });
    });



