const fs = require('fs-extra');

const readJsonFile = async (file_directory) => {
    const myJsonObject = await fs.readJson(file_directory);
    console.log(myJsonObject);
}

export default readJsonFile;