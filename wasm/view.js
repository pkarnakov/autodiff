var debug = false;
var textarea_out;
var textarea_err;
var g_shared_canvas;

// Colors.
var c_red = "#ff1f5b";
var c_green = "#00cd6c";
var c_blue = "#009ade";
var c_orange = "#f28522";
var c_gray = "#a0b1ba";
var c_white = "#ffffff";
var c_black = "#000000";
var c_background = "#50585d";

// Control.
var SendKeyDown;
var SendMouseMotion;
var SendMouseDown;
var SendMouseUp;

// Misc.
var SetPause;
var GetStatusString;
var GetBitmapWidth;
var GetBitmapHeight;
var flag_pause = false;
var Init;

function draw() {
  let canvas = Module['canvas'];
  let ctx = canvas.getContext('2d');
  ctx.drawImage(g_shared_canvas, 0, 0, canvas.width, canvas.height);
  window.text_status.innerHTML = GetStatusString();
}

function restart() {
  Init();
  flag_pause = false;
  flag_accel = false;
  syncButtons();
}

function setButtonStyle(button_name, pressed) {
  document.getElementById('button_' + button_name).className =
      pressed ? 'button pressed' : 'button';
}

function syncButtons() {
  setButtonStyle('pause', flag_pause);
}

function togglePause() {
  flag_pause = !flag_pause;
  syncButtons();
  SetPause(flag_pause);
}

function clearOutput() {
  if (textarea_out) {
    textarea_out.value = '';
  }
  if (textarea_err) {
    textarea_err.value = '';
  }
}
function printOut(text) {
  if (textarea_out) {
    textarea_out.value += text + "\n";
    textarea_out.scrollTop = textarea_out.scrollHeight;
  }
}
function printErr(text) {
  if (textarea_err) {
    textarea_err.value += text + "\n";
    textarea_err.scrollTop = textarea_err.scrollHeight;
  }
}

function sendKeyDownChar(c) {
  keysym = c.charCodeAt(0);
  if (keysym < 256) {
    SendKeyDown(keysym);
  }
}

function pressButton(c) {
  if (c == 'r') {
    restart();
  } else {
    sendKeyDownChar(c);
    syncButtons();
  }
}

function preRun() {
  if (debug) {
    const fragment = document.createDocumentFragment();
    const div = fragment.appendChild(document.createElement("div"));
    div.className = "row";
    textarea_out = document.createElement("textarea")
    textarea_out.id = "textarea_out";
    div.appendChild(textarea_out);
    textarea_err = document.createElement("textarea")
    textarea_err.id = "textarea_err";
    div.appendChild(textarea_err);
    window.content_column.appendChild(fragment);
  }
}

function postRun() {
  SendKeyDown = Module.cwrap('SendKeyDown', null, ['number']);
  SendMouseMotion = Module.cwrap('SendMouseMotion', null, ['number', 'number']);
  SendMouseDown = Module.cwrap('SendMouseDown', null, ['number', 'number']);
  SendMouseUp = Module.cwrap('SendMouseUp', null, ['number', 'number']);
  SetPause = Module.cwrap('SetPause', null, ['number']);
  GetStatusString = Module.cwrap('GetStatusString', 'string', []);
  GetBitmapWidth = Module.cwrap('GetBitmapWidth', 'number', []);
  GetBitmapHeight = Module.cwrap('GetBitmapHeight', 'number', []);
  Init = Module.cwrap('Init', null, []);

  let canvas = Module['canvas'];
  g_shared_canvas = document.createElement('canvas');
  g_shared_canvas.width = GetBitmapWidth();
  g_shared_canvas.height = GetBitmapHeight();

  let handler_keydown = function(e) {
    let nodename = e.target.nodeName;
    if (nodename == 'TEXTAREA') {
      return;
    }
    if (e.key == ' ') {
      e.preventDefault();
      togglePause();
    }
    pressButton(e.key);
  };
  let get_xy = function(e) {
    let x = e.offsetX / canvas.clientWidth;
    let y = 1 - e.offsetY / canvas.clientHeight;
    return [x, y]
  };
  let get_touch_xy = function(e, touch) {
    var rect = e.target.getBoundingClientRect();
    var x = (e.changedTouches[0].pageX - rect.left) / canvas.clientWidth;
    var y = 1 - (e.changedTouches[0].pageY - rect.top) / canvas.clientHeight;
    return [x, y]
  };


  let handler_mousemove = function(e) {
    e.preventDefault();
    xy = get_xy(e);
    SendMouseMotion(xy[0], xy[1]);
  };
  let handler_mousedown = function(e) {
    e.preventDefault();
    xy = get_xy(e);
    SendMouseDown(xy[0], xy[1]);
  };
  let handler_mouseup = function(e) {
    e.preventDefault();
    xy = get_xy(e);
    SendMouseUp(xy[0], xy[1]);
  };
  let handler_touchmove = function(e) {
    e.preventDefault();
    xy = get_touch_xy(e);
    SendMouseMotion(xy[0], xy[1]);
  };
  let handler_touchstart = function(e) {
    e.preventDefault();
    xy = get_touch_xy(e);
    SendMouseDown(xy[0], xy[1]);
  };
  let handler_touchend = function(e) {
    e.preventDefault();
    xy = get_touch_xy(e);
    SendMouseUp(xy[0], xy[1]);
  };

  window.addEventListener('keydown', handler_keydown);
  canvas.addEventListener('mousemove', handler_mousemove);
  canvas.addEventListener('mousedown', handler_mousedown);
  canvas.addEventListener('mouseup', handler_mouseup);
  canvas.addEventListener('touchmove', handler_touchmove);
  canvas.addEventListener('touchstart', handler_touchstart);
  canvas.addEventListener('touchend', handler_touchend);

  syncButtons();
}

var Module = {
  preRun: [preRun],
  postRun: [postRun],
  print: (function(text) {
    clearOutput();
    return function(text) {
      if (arguments.length > 1) {
        text = Array.prototype.slice.call(arguments).join(' ');
      }
      printOut(text);
    };
  })(),
  printErr: (function(text) {
    clearOutput();
    return function(text) {
      if (arguments.length > 1) {
        text = Array.prototype.slice.call(arguments).join(' ');
      }
      printErr(text);
    };
  })(),
  canvas: (function() { return document.getElementById('canvas'); })(),
  setStatus: function(text) {},
};
