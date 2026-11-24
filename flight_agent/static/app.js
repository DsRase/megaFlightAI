const root = document.documentElement;
const themeButton = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const themeLabel = document.getElementById('theme-label');
const form = document.getElementById('chat-form');
const input = document.getElementById('message-input');
const sendButton = document.getElementById('send-button');
const clearButton = document.getElementById('clear-button');
const messages = document.getElementById('messages');
const error = document.getElementById('error');
const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
const thoughts = [
  'Проверяю маршрут', 'Сверяюсь с картой', 'Ищу подходящий ответ',
  'Изучаю варианты', 'Прокладываю курс', 'Собираю информацию',
  'Проверяю детали', 'Смотрю, что можно найти', 'Думаю над вашим вопросом',
  'Готовлю ответ', 'Уточняю направление', 'Планирую следующий шаг'
];
let pending = false;

function setTheme(theme) {
  root.dataset.theme = theme;
  const dark = theme === 'dark';
  themeIcon.textContent = dark ? '☀' : '☾';
  themeLabel.textContent = dark ? 'Светлая тема' : 'Тёмная тема';
  themeButton.setAttribute('aria-label', `Включить ${dark ? 'светлую' : 'тёмную'} тему`);
}

let savedTheme;
try { savedTheme = localStorage.getItem('theme'); } catch (_) { /* storage may be unavailable */ }
setTheme(savedTheme === 'dark' || savedTheme === 'light' ? savedTheme : (systemTheme.matches ? 'dark' : 'light'));
themeButton.addEventListener('click', () => {
  const theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
  setTheme(theme);
  try { localStorage.setItem('theme', theme); } catch (_) { /* keep the current session theme */ }
});

async function serverReply(response) {
  const data = await response.json();
  if (!data || typeof data.reply !== 'string') throw new Error('response');
  return data.reply;
}

function maskPassports(text) {
  return text
    .replace(/(?<![\p{L}\p{N}_])((?:паспорт(?:а|у|ом)?|passport)(?![\p{L}\p{N}_])\s*[":=№#-]*\s*)(?!\d{4}-\d{2}-\d{2}(?![\p{L}\p{N}_]))([0-9]+(?:[ /-]+[0-9]+)*|[A-Za-z]+(?:\/[A-Za-z0-9]+|[0-9][A-Za-z0-9]*)+)(?![\p{L}\p{N}_-])/giu, '$1***')
    .replace(/(?<![\p{L}\p{N}_-])(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d{10})[A-Za-z0-9]+(?![\p{L}\p{N}_-])/gu, '***')
    .replace(/(?<![\p{L}\p{N}_-])(?:\d{10,}|\d{4}[ -]+\d{6}|\d{2}[ -]+\d{2}[ -]+\d{6})(?![\p{L}\p{N}_-])/gu, '***');
}

function setPending(value) {
  pending = value;
  sendButton.disabled = value;
  clearButton.disabled = value;
}

clearButton.addEventListener('click', async () => {
  if (pending) return;
  setPending(true);
  error.hidden = true;
  error.textContent = '';
  try {
    const response = await fetch('/api/chat/clear', { method: 'POST' });
    const reply = await serverReply(response);
    if (!response.ok) { error.textContent = reply; error.hidden = false; return; }
    messages.replaceChildren();
    input.value = '';
    resizeInput();
  } catch (_) {
    error.textContent = 'Не удалось получить результат очистки. Проверьте состояние чата.';
    error.hidden = false;
  } finally {
    setPending(false);
  }
});

function addMessage(text, fromUser = false) {
  const row = document.createElement('div');
  row.className = `message ${fromUser ? 'user-message' : 'assistant-message'}`;
  if (!fromUser) {
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.setAttribute('aria-hidden', 'true');
    avatar.textContent = '✳';
    row.append(avatar);
  }
  const content = document.createElement('div');
  content.className = 'message-content';
  const author = document.createElement('span');
  author.className = 'message-author';
  author.textContent = fromUser ? 'ВЫ' : 'АВИАПОМОЩНИК';
  const bubble = document.createElement('p');
  bubble.className = 'bubble';
  bubble.textContent = text;
  content.append(author, bubble);
  row.append(content);
  messages.append(row);
  messages.scrollTop = messages.scrollHeight;
  return { row, bubble };
}

function resizeInput() {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
}
input.addEventListener('input', resizeInput);
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const message = input.value;
  if (!message.trim() || pending) return;
  setPending(true);
  error.hidden = true;
  error.textContent = '';
  addMessage(maskPassports(message), true);
  input.value = '';
  resizeInput();
  const thinking = addMessage(thoughts[Math.floor(Math.random() * thoughts.length)]);
  const symbol = document.createElement('span');
  symbol.className = 'thinking-symbol';
  symbol.setAttribute('aria-hidden', 'true');
  symbol.textContent = ' ✦';
  thinking.bubble.append(symbol);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180000);
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
      signal: controller.signal
    });
    const reply = await serverReply(response);
    thinking.row.remove();
    if (!response.ok) { error.textContent = reply; error.hidden = false; return; }
    addMessage(reply);
  } catch (err) {
    thinking.row.remove();
    error.textContent = err.name === 'AbortError'
      ? 'Время ожидания истекло. Проверьте результат перед повторной отправкой.'
      : err instanceof TypeError
      ? 'Не удалось связаться с сервером. Результат неизвестен; проверьте состояние броней перед повторной отправкой.'
      : 'Не удалось получить ответ. Результат неизвестен; проверьте состояние броней перед повторной отправкой.';
    error.hidden = false;
  } finally {
    clearTimeout(timeout);
    setPending(false);
  }
});
