function next_week(curr_date, no_of_week) {
    let total_days = no_of_week * 7;

    let today = new Date(curr_date);
    let next_week_date = new Date(today.getFullYear(), today.getMonth(), today.getDate() + total_days).getTime();
    return next_week_date;
}

export {next_week};


