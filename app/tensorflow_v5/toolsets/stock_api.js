import fetch from "node-fetch";

/* this api gets all the stock names stored in the server */
async function get_stock_list() {
    return await fetch("https://algotuno-web3.vercel.app/api/stock/get_all_stocks").then(res => {
        // .map(stock_obj => stock_obj['tickerSymbol'].lower)
        return res.json().then(res => res.result.map(x => x['tickerSymbol']));
    });
}


export {
    get_stock_list
};