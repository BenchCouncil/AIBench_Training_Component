'use strict';

module.exports = function (str, prefix) {

  if (typeof prefix === 'undefined' || prefix === null) {
    return false;
  }

  str    = String(str);
  prefix = String(prefix);

  var l = prefix.length;
  var i = -1;

  while (++i < l) {
    if (str.charAt(i) !== prefix.charAt(i)) {
      return false;
    }
  }

  return true;
};
