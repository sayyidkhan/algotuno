// https://dev.to/ramonak/javascript-how-to-access-the-return-value-of-a-promise-object-1bck
const fs = require('fs-extra');

const readJsonFile = async (file_directory) => {
    const myJsonObject = await fs.readJson(file_directory);
    return myJsonObject;
}

function extract_close_and_date_only(stock_json, year) {
    stock_json[year] = stock_json[year].map(day_record => {
        return {
            'Date': day_record.Date,
            'Close': day_record.Close,
        }
    });
}


async function main() {
    let google = {};
    //1. get data
    await readJsonFile("./google_2017.json").then(res => google['2017'] = res);
    await readJsonFile("./google_2018.json").then(res => google['2018'] = res);
    //2. extract date and close price only
    extract_close_and_date_only(google, '2017');
    extract_close_and_date_only(google, '2018');
    console.log(google);
}


main();
