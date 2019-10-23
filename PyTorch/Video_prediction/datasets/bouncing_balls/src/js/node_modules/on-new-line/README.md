# on-new-line

> Hijacks NodeJS Stream and emits events when newline(s) written to the output.

[![MIT License](https://img.shields.io/badge/license-MIT_License-green.svg?style=flat-square)](https://github.com/bubkoo/on-new-line/blob/master/LICENSE)
[![npm:](https://img.shields.io/npm/v/on-new-line.svg?style=flat-square)](https://www.npmjs.com/packages/on-new-line)
 

## Install

```
$ npm install on-new-line --save
```

## Usage

```js
require('on-new-line')(stream); // Any type of NodeJS stream, such as: filestream, process.stdout etc.

// emitted on every newline detected
stream.on('newline', function() {

});

// emitted before data written to the output. 
stream.on('before:newlines', function(lineCount) {

});

// emitted after data written to the output. 
stream.on('after:newlines', function(lineCount) {

});

```


## Contributing

Pull requests and stars are highly welcome.

For bugs and feature requests, please [create an issue](https://github.com/bubkoo/on-new-line/issues/new).
