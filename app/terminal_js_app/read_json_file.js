const fs = require('fs');

let rawdata = fs.readFileSync('appl_price.json');
let stock_data = JSON.parse(rawdata);

const clone = (items) => items.map(item => Array.isArray(item) ? clone(item) : item);

const predict_up_down_hold = (curr_price, future_price) => {
    // if price 116.52 and 116. 32 (small price movement) - HOLD
    if (parseInt(curr_price) === parseInt(future_price)) {
        return 0;
    }
    // if curr price greater than buying price - DOWN
    else if (parseInt(curr_price) > parseInt(future_price)) {
        return -1;
    }
    // if curr price lesser than buying price - UP
    else {
        return 1;
    }
};

const extract_data = (_data) => {
    return _data.map(obj => {
        return {
            'Date': obj.Date,
            'Epoch': new Date(obj.Date).getTime(),
            'Close': parseFloat(parseFloat(obj.Close).toFixed(2)),
            'Predict': undefined,
            'Next': undefined,
        };
    });

};

const filter_year = (_data, year) => {
    return _data.map(x => x).filter(obj => {
        return new Date(obj.Date).getUTCFullYear() === year
    });
};

const group_according_to_month = (_data) => {
    // basic group either within 1, 7 , 30 days
    const year_list = [];
    // init 12 arrays in year list, for each and every month
    for (let i = 0; i < 12; i++) {
        year_list.push([]);
    }

    // group the data according to month
    _data.forEach(x => {
        const month = new Date(x.Date).getMonth(); // month is alreadly at zero based index where month 1 is 0 in index
        const month_list = year_list[month];
        month_list.push(x);
    });
    return year_list;
};

const get_next_wk_record_in_curr_record = (_data) => {
    for (let month = 0; month < _data.length - 1; month++) {
        const month_record = _data[month];
        const next_month_record = _data[month + 1];

        const next_month_first_record = next_month_record[0];
        for (let week = 0; week < month_record.length; week++) {
            const curr_week = month_record[week];

            // at the last record, take the next month record
            if (week === month_record.length - 1) {
                const _predict = predict_up_down_hold(curr_week.Close, next_month_first_record.Close);
                curr_week['Next'] = next_month_first_record.Close;
                curr_week['Predict'] = _predict;
            } else {
                const next_week = month_record[week + 1];
                const _predict = predict_up_down_hold(curr_week.Close, next_week.Close);
                curr_week['Next'] = next_week.Close;
                curr_week['Predict'] = _predict;
            }
        }
    }
    return _data;
};


function get_stock_data(year) {
// NOTE: this dataset only have the date over a 7 day period - so it can only predict every 7 days

// 1. extract the Date and the close price and create the epoch object
    stock_data = extract_data(stock_data);
// 2. filter the curr year
    let stock_data_curr = filter_year(stock_data, year);
// 3. filter the next year
    let stock_data_next = filter_year(stock_data, year + 1);
// 4. put the next week record in the current obj
    let group_mth_stock_data_curr = group_according_to_month(stock_data_curr);
    let group_mth_stock_data_next = group_according_to_month(stock_data_next);


    group_mth_stock_data_curr.push(group_mth_stock_data_next[0]); // later aft processing will drop the 13th month
//console.log(group_mth_stock_data_curr);

    get_next_wk_record_in_curr_record(group_mth_stock_data_curr);
    group_mth_stock_data_curr.pop(); // remove 13 mth record
    console.log(group_mth_stock_data_curr);
    return group_mth_stock_data_curr;
}

function prepare_stock_data(_data) {
    // data is a list of list - so need to loop through it twice
    const xTraining = [];
    const yTraining = [];

    _data.forEach(month_list => {
        const xMonthList = [];
        const yMonthList = [];
        month_list.forEach(week => {
            const weeklyrecord_x = [week.Epoch, week.Close];
            const weeklyrecord_y = [week.Predict, week.Next];
            xMonthList.push(weeklyrecord_x);
            yMonthList.push(weeklyrecord_y);
        });
        // add the month list to the training list
        xTraining.push(xMonthList);
        yTraining.push(yMonthList);
    });

    return {
        'x': xTraining,
        'y': yTraining,
    };
}

const stock_data2016 =get_stock_data(2016);
const p_data2016 = prepare_stock_data(stock_data2016);
console.log(p_data2016.x);