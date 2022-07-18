function meanAbsolutePercentageError(forecast, actual) {
    const n = forecast.length;
    if (actual.length !== n) throw new Error("forceast and actual must be same length"); // <-- what to do here?
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += Math.abs((actual[i] - forecast[i]) / actual[i]);
    }
    return sum * 100 / n;
}


function rebase_to_one(_my_list) {
    /* this function is used to make the value in the list if the value is lesser than 1, to return 1
    otherwise return the current value it is at
     */
    return _my_list.map(my_int => {
        if (my_int >= 1) {
            return my_int;
        } else {
            return 1;
        }
    })
}
