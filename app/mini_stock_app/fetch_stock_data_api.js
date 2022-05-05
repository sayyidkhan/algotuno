const fetch = require('node-fetch');

async function get_stock_data() {
    const url = 'https://algotuno-web.vercel.app/api/stock/get_hsp';
    let response = await fetch(url, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json;charset=utf-8'
        },
        body: JSON.stringify({
            "ticker_symbol" : 	"GOOG",
            "start_date"	:	"2017-01-01",
            "end_date"	:	"2017-12-31",
            "sort"		:	"asc"
        })
    });
    return response;
}


get_stock_data().then(res => {
    console.log(res);
});