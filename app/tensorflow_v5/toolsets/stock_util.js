import fs from "fs";


function get_stock_data_from_fs(_dir) {
    /*
    1. read the stock data from File system
    2. extract the relevant data from the stock data
     */
    function read_json_file(_dir) {
        /*
        used to get the stock data from the file system
         */
        let _raw_data = fs.readFileSync(_dir);
        return JSON.parse(_raw_data);
    }

    return read_json_file(_dir);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export {get_stock_data_from_fs, sleep};