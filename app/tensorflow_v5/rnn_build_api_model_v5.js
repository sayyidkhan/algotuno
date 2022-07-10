import {get_price_list_for_many_stocks, get_stock_list} from "./toolsets/stock_api.js";
import {run_build_model} from "./rnn_build_model_v5.js";
import * as fsExtra from "fs-extra";

/*** variables ***/
const should_delete_all_existing_files = false;
const should_rebuild_model = true;

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


// 1. delete all existing files
if (should_delete_all_existing_files) {
    delete_all_existing_files().then(res => { console.log("--- ALL FILES DELETED SUCCESSFULLY ---") })
}

// 2. rebuild model
if (should_rebuild_model) {
    build_api_model_v5().then(res => {console.log("--- RE-BUILD MODEL COMPLETED ---")});
}
