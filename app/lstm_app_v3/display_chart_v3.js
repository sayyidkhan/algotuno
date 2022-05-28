function get_accuracy(_validation_dataset, _prediction_dataset) {
    /***
     *
     * we are going to skip the first data as we do not have the previous number to confirm the
     * prediction whether the price went up or down
     *
     * ***/
    const get_direction = (_prev, _curr) => {
        if (_curr > _prev) {
            return "UP";
        } else {
            return "DOWN";
        }
    };

    let score = 0;
    for (let index = 1; index < _validation_dataset.length; index++) {
        const prev_v_data = _validation_dataset[index - 1];
        const v_data = _validation_dataset[index];
        const v_direction = get_direction(prev_v_data, v_data);

        const prev_p_data = _prediction_dataset[index - 1];
        const p_data = _prediction_dataset[index];
        const p_direction = get_direction(prev_p_data, p_data);

        if (v_direction === p_direction) {
            score += 1;
        }
        //const p_data = _prediction_dataset[index];
    }

    // accuracy rate in percentage
    const in_percentage = parseFloat(((score / _validation_dataset.length) * 100).toFixed(2));
    const in_number = `${score} / ${_validation_dataset.length}`;
    return {
        "in_percentage": in_percentage,
        "in_number": in_number,
    }
}

export {
    get_accuracy
}