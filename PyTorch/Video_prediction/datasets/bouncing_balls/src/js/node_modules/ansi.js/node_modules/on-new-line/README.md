# on-new-line
Hijacks NodeJS Stream and emits events when newline(s) written to the output.


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
