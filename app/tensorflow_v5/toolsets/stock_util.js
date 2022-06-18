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
        let rawdata = fs.readFileSync(_dir);
        return JSON.parse(rawdata);
    }

    return read_json_file(_dir);
}


export {get_stock_data_from_fs};