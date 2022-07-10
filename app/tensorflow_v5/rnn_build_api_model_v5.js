import {get_price_list_for_many_stocks, get_stock_list} from "./toolsets/stock_api.js";

/*** variables ***/

async function build_api_model_v5() {
    /*** 1. fetch the list of stock names ***/
    const list_of_stocks = await get_stock_list().then(res => res);
    console.log("1 . GET list of stocks");
    console.log(list_of_stocks);
    /*** 2. fetch the stock data ***/
    const fetched_price_list = await get_price_list_for_many_stocks(list_of_stocks);
    console.log(fetched_price_list);
    /*** 3. train the model and output the train model into `tf_models_automation` directory ***/
}


build_api_model_v5().then(res => {
    console.log("--- JOB COMPLETED ---");
});
