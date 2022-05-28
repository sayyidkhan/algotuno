let x1 = [39.48, 35.55, 37.06, 36.98];
let y1 = [37.69];
let original_x1 = [
    [39.48, 1546387200000],
    [35.55, 1546473600000],
    [37.06, 1546560000000],
    [36.98, 1546819200000]
];
let original_y1 = [[37.69, 1546905600000],];


console.log(original_x1.map(x => [x[0], new Date(x[1])]));
console.log(original_y1.map(x => [x[0], new Date(x[1])]));