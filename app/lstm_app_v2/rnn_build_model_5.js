import {get_stock_data_from_fs} from "./stock_util_v2.js";
import {prepare_xs_ys_dataset, nn_model, predict_v, train} from "./main_v2.js";
import {plot} from "nodeplotlib";


/*** variables ***/
const _dir_to_save_file = 'file://tf_models/lstm_5';
const _dir_stock_data = './stock_price/appl_price.json';
const window = 2; // every 2 weeks (each whole no represent every week)
const epoch = 10000; // no of times to train
const display_training_output = true; // display logs on training ml_output per epoch
const display_model_summary = false;


/*** program logic ***/

// 1. get training and labelled dataset
let input_stock_data = get_stock_data_from_fs(_dir_stock_data);
const training_dataset_2016 = prepare_xs_ys_dataset(input_stock_data, 2016, window);
const validation_dataset_2017 = prepare_xs_ys_dataset(input_stock_data, 2017, window);
const test_dataset_2018 = prepare_xs_ys_dataset(input_stock_data, 2018, window);

// 2. setup the neural network model
const sample_model = nn_model(
    training_dataset_2016.tensor_xs_data,
    training_dataset_2016.tensor_ys_data
);


const ml_output = async () => {
    // 3. train the model
    let train_output = await train(
        training_dataset_2016.tensor_xs_data,
        training_dataset_2016.tensor_ys_data,
        epoch,
        sample_model,
        display_training_output,
    ).then((_training_output) => {
        // save model
        if (display_model_summary) {
            console.log(sample_model.summary());
        }
        sample_model.save(_dir_to_save_file);
        console.log("model saved successfully");

        return _training_output.loss_output;
    });
    // 4. predict validation result
    /*** validation dataset result ***/
    const validation_result = await predict_v(
        sample_model,
        validation_dataset_2017.tensor_xs_data,
        validation_dataset_2017.original_ys_data
    );

    console.log(validation_result);

};

ml_output();


