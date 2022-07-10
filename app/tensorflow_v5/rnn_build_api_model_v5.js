import {get_price_list_for_many_stocks, get_stock_list} from "./toolsets/stock_api.js";
import {run_build_model} from "./rnn_build_model_v5.js";
import {get_file_from_fs} from "./toolsets/stock_util.js";
import * as fsExtra from "fs-extra";
import {upload_model} from "./toolsets/azure_upload_file.js";
import {getDirectories} from "./toolsets/stock_util.js";


/*** variables ***/
const should_delete_all_existing_files = false;
const should_rebuild_model = false;
const should_upload_model = true;

/*** delete all existing files ***/
async function delete_all_existing_files() {
    fsExtra.emptyDirSync('./tf_models_automation/');
}

/*** rebuild model function ***/
async function build_api_model_v5() {
    /*** 1. fetch the list of stock names ***/
    const list_of_stocks = await get_stock_list().then(res => res);
    console.log("1 . GET list of stocks");
    console.log(list_of_stocks);
    /*** 2. fetch the stock data ***/
    const fetched_price_list = await get_price_list_for_many_stocks(list_of_stocks);
    // console.log(fetched_price_list);
    /*** 3. train the model and output the train model into `tf_models_automation` directory ***/
    const _dir_to_save_file = (_stock_name) => `file://tf_models_automation/${_stock_name.toLowerCase()}`;
    for (const stock_entity_index in fetched_price_list) {
        const stock_entity = fetched_price_list[stock_entity_index];
        const _stock_name = stock_entity.stock_name;
        const _list = stock_entity.stock_list;
        const file_dir = _dir_to_save_file(_stock_name);
        run_build_model(_stock_name, _list, file_dir);
    }
}

/*** upload model function ***/
async function upload_model_to_azure() {
    /*** 1. get all the files in the `tf_models_automation` ***/
    const file_directory = "./tf_models_automation/";
    let dir_list = getDirectories(file_directory);
    // comment out, to run the production upload
    dir_list = dir_list.filter(x => x === "test_tsla");
    console.log("list of files to be uploaded: ");
    console.log(dir_list);

    /*** 2. upload all the file set one by one into the cloud ***/
    for (const file_name_index in dir_list) {
        const _stock_name = dir_list[file_name_index];
        const file_model = get_file_from_fs(`./tf_models_automation/${_stock_name}/model.json`);
        const file_weights = get_file_from_fs(`./tf_models_automation/${_stock_name}/weights.bin`);

        // upload file to azure
        await upload_model(_stock_name, file_model, file_weights).then(res => {
            console.log(res);
        }).catch(err => {
            console.log(err);
        })
    }
}


// 1. delete all existing files
if (should_delete_all_existing_files) {
    delete_all_existing_files().then(res => {
        console.log("--- ALL FILES DELETED SUCCESSFULLY ---")
    })
}

// 2. rebuild model
if (should_rebuild_model) {
    build_api_model_v5().then(res => {
        console.log("--- RE-BUILD MODEL COMPLETED ---")
    });
}

// 3. upload model
if (should_upload_model) {
    upload_model_to_azure('test_tsla').then(res => res);
}

