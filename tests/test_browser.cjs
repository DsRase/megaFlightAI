// Run separately with: node --test tests/test_browser.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function browser(fetch, storage) {
  class Element {
    constructor() {
      this.children = [];
      this.listeners = {};
      this.style = {};
      this.dataset = {};
      this.value = '';
      this.textContent = '';
    }
    addEventListener(type, listener) { this.listeners[type] = listener; }
    setAttribute() {}
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    remove() { this.removed = true; }
  }
  const ids = Object.fromEntries(['theme-toggle', 'theme-icon', 'theme-label', 'chat-form',
    'message-input', 'send-button', 'clear-button', 'messages', 'error'].map(id => [id, new Element()]));
  const document = {documentElement: new Element(), getElementById: id => ids[id], createElement: () => new Element()};
  const context = {document, window: {matchMedia: () => ({matches: false})},
    fetch, AbortController, setTimeout, clearTimeout, localStorage: storage};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../flight_agent/static/app.js'), 'utf8'), context);
  ids.root = document.documentElement;
  return ids;
}

test('theme preference persists and server errors are displayed safely', async () => {
  const values = {};
  const storage = {getItem: key => values[key], setItem: (key, value) => { values[key] = value; }};
  const fetch = async () => ({ok: false, json: async () => ({reply: 'Приложение занято.'})});
  const ids = browser(fetch, storage);
  ids['theme-toggle'].listeners.click();
  assert.equal(values.theme, 'dark');
  assert.deepEqual(Object.keys(values), ['theme']);
  assert.equal(browser(fetch, storage).root.dataset.theme, 'dark');
  ids['message-input'].value = 'Бронь';
  await ids['chat-form'].listeners.submit({preventDefault() {}});
  assert.equal(ids.error.textContent, 'Приложение занято.');
});

test('shows user text safely and unchanged', async () => {
  const ids = browser(async () => ({ok: true, json: async () => ({reply: 'Готово'})}));
  ids['message-input'].value = '  <b>Привет</b>! Паспорт: 1234 567890, рейс SU123.  ';
  await ids['chat-form'].listeners.submit({preventDefault() {}});
  const userText = ids.messages.children[0].children[0].children[1].textContent;
  assert.equal(userText, '  <b>Привет</b>! Паспорт: 1234 567890, рейс SU123.  ');
});

test('keeps flight details and ordinary passport questions readable', async () => {
  const ids = browser(async () => ({ok: true, json: async () => ({reply: 'Готово'})}));
  const message = 'Рейс SU202 2027-06-01 12:40, 220 рейсов, 123 места. Паспортные данные: а паспорт какой? passport please, паспорт 2027-06-12, предпаспорт 987, паспортный 987. 1234567890';
  ids['message-input'].value = message;
  await ids['chat-form'].listeners.submit({preventDefault() {}});
  assert.equal(ids.messages.children[0].children[0].children[1].textContent, message);
});

test('shows labeled values and separated passport numbers unchanged', async () => {
  const ids = browser(async () => ({ok: true, json: async () => ({reply: 'Готово'})}));
  ids['message-input'].value = 'паспорт 987, passport ABC/DEF, паспорт 1234 567890, паспорт 12 34 567890';
  await ids['chat-form'].listeners.submit({preventDefault() {}});
  assert.equal(ids.messages.children[0].children[0].children[1].textContent, 'паспорт 987, passport ABC/DEF, паспорт 1234 567890, паспорт 12 34 567890');
});

test('clear waits for active requests and clears history only on success', async () => {
  let finishChat;
  let finishClear;
  const calls = [];
  const ids = browser((url, options) => {
    calls.push([url, options.method]);
    return new Promise(resolve => {
      if (url === '/api/chat') finishChat = resolve;
      else finishClear = resolve;
    });
  });
  ids['message-input'].value = 'Привет';
  const chat = ids['chat-form'].listeners.submit({preventDefault() {}});
  assert.equal(ids['clear-button'].disabled, true);
  await ids['clear-button'].listeners.click();
  assert.equal(calls.length, 1);
  finishChat({ok: true, json: async () => ({reply: 'Здравствуйте'})});
  await chat;
  const clear = ids['clear-button'].listeners.click();
  assert.equal(ids['send-button'].disabled, true);
  ids['message-input'].value = 'Ещё вопрос';
  await ids['chat-form'].listeners.submit({preventDefault() {}});
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[1], ['/api/chat/clear', 'POST']);
  finishClear({ok: false, json: async () => ({reply: 'Занято'})});
  await clear;
  assert.ok(ids.messages.children.length);
  assert.equal(ids.error.hidden, false);
  const retry = ids['clear-button'].listeners.click();
  finishClear({ok: true, json: async () => ({reply: 'Очищено'})});
  await retry;
  assert.equal(ids.messages.children.length, 0);
  assert.equal(ids['message-input'].value, '');
  assert.equal(ids['send-button'].disabled, false);
});

function contents(element) {
  return [element.textContent, element.value, ...element.children.map(contents)].join(' ');
}

for (const failure of [false, true]) {
  test(`passport stays in the user message and input is cleared after ${failure ? 'failure' : 'success'}`, async () => {
    let sent;
    const ids = browser(async (_url, options) => {
      sent = JSON.parse(options.body).message;
      assert.ok(options.signal);
      if (failure) throw new Error('private 1234567890');
      return {ok: true, json: async () => ({reply: 'Готово'})};
    });
    ids['message-input'].value = 'Паспорт 1234567890';
    await ids['chat-form'].listeners.submit({preventDefault() {}});
    assert.equal(sent, 'Паспорт 1234567890');
    assert.equal(ids['message-input'].value, '');
    assert.equal(ids.messages.children[0].children[0].children[1].textContent, 'Паспорт 1234567890');
    assert.doesNotMatch(contents(ids.error), /1234567890/);
    assert.equal(ids['send-button'].disabled, false);
  });
}
