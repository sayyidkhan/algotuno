import {get_price_list_for_many_stocks, get_stock_list} from "./toolsets/stock_api.js";
import {run_build_model} from "./rnn_build_model_v5.js";

/*** variables ***/

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


build_api_model_v5().then(res => {
    console.log("--- JOB COMPLETED ---");
});
