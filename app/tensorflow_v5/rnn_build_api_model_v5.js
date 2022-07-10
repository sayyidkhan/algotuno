import { get_stock_list } from "./toolsets/stock_api.js";

/*** variables ***/

async function build_api_model_v5() {
    /*** 1. fetch the list of stock names ***/
    const response = await get_stock_list().then(res => res);
    console.log(response);


    /*** 2. fetch the stock data ***/

    /*** 3. train the model and output the train model into `tf_models_automation` directory ***/
}


build_api_model_v5().then(res => {
   console.log("--- JOB COMPLETED ---");
});
