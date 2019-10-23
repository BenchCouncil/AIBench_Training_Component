/**
 * Accepts any node Stream instance and hijacks its `write()` function,
 * so that it can count any newlines that get written to the output.
 *
 * When a '\n' byte is encountered, then a "newline" event will be emitted
 * on the stream, with no arguments. It is up to the listeners to determine
 * any necessary deltas required for their use-case.
 */


function emitNewlineEvent(stream) {

  if (stream._newlineEventInsalled) {
    // already emitting newline events
    return;
  }

  var write = stream.write;

  stream.write = function (data) {

    var counter = 0;

    if (stream.listeners('newline').length
      || stream.listeners('before:newlines').length
      || stream.listeners('after:newlines').length) {

      var i = 0;
      var l = data.length;

      // now try to calculate any deltas
      if (typeof data === 'string') {
        for (; i < l; i++) {
          if (processByte(stream, data.charCodeAt(i))) {
            counter += 1;
          }
        }
      } else {
        // buffer
        for (; i < l; i++) {
          if (processByte(stream, data[i])) {
            counter += 1;
          }
        }
      }
    }

    if (counter) {
      stream.emit('before:newlines', counter);
    }

    // write the data
    var result = write.apply(stream, arguments);

    if (counter) {
      stream.emit('after:newlines', counter);
    }

    return result
  };

  stream._newlineEventInsalled = true;
}


// exports
// -------

module.exports = emitNewlineEvent;


// helpers
// -------

var NEWLINE = '\n'.charCodeAt(0);

function processByte(stream, byte) {

  // processes an individual byte being written to a stream

  if (typeof byte !== 'number') {
    throw new Error('Invalid byte data for stream.');
  }

  if (byte === NEWLINE) {
    stream.emit('newline');
    return true;
  }
}
