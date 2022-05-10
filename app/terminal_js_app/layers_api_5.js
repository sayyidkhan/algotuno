const tf = require('@tensorflow/tfjs-node');
const {plot, Plot} = require('nodeplotlib');

const model = tf.sequential();

// dense is a fully connected layer
const hidden = tf.layers.inputLayer({
    units: 1, // number of nodes
    inputShape: [4], // input shape
});
model.add(hidden);

// const hidden_2 = tf.layers.dense({
//     units: (4 * 3),
//     // here the input shape is "inferred from the previous shape"
//     activation: 'elu',
// });
// model.add(hidden_2);
//
// const hidden_3 = tf.layers.dense({
//     units: (4 * 3),
//     // here the input shape is "inferred from the previous shape"
//     activation: 'tanh',
// });
// model.add(hidden_3);


const output = tf.layers.dense({
    units: 1,
    // here the input shape is "inferred from the previous shape"
    activation: 'linear',
});
model.add(output);

// i'm done configuring the model so compile it
model.compile({
    optimizer: tf.train.adam(0.00005),
    loss: tf.losses.meanSquaredError,
});


// INPUT DATA
const xs = tf.tensor2d([
    [96.96, 97.13, 101.42, 97.34],
]);

// OUTPUT DATA
const ys = tf.tensor2d(
    [
        [94.02]
    ]);

//console.log(xs.shape);
console.log(xs);
console.log(model.summary());

async function train() {

    for (let i = 0; i < 1000; i++) {
        const config = {
            epochs: 10,
        };
        const response = await model.fit(xs, ys, config);
        console.log(response.history.loss[0]);
    }

}

function convert_to_x_and_y_axis(xs_list) {
    let x = [];
    let y = [];
    for (var i = 0; i < xs_list.length; i++) {
        var curr_item = xs_list[i];

        x.push(i + 1);
        y.push(curr_item);
    }

    let obj = {
        'x-axis': x,
        'y-axis': y,
    }
    return obj;
}

function generate_plot_line(price_list, name) {
    var obj = convert_to_x_and_y_axis(price_list);
    return {x: obj['x-axis'], y: obj['y-axis'], type: 'plot', name: name};
}


// // do the train first before doing the prediction
train()
    .then(() => {
        console.log("training complete");
        let outputs = model.predict(xs);
        return outputs;
    })
    .then((_outputs) => {
        console.log("tensor predict output");
        console.log(_outputs.print());
    });
// let _predict = _outputs.arraySync().map(x => x[0]);
// console.log(_predict);
// console.log("display result");
// _predict = generate_plot_line(_predict, "predict");
//
// let _actual = generate_plot_line(ys.arraySync().map(x => x[0]), "actual");
// return {
//     'actual': _actual,
//     'predict': _predict,
// }

// .then((_predict_obj) => {
//     const actual_result = _predict_obj.actual;
//     const predict_result = _predict_obj.predict;
//     plot([actual_result, predict_result]);
// });





