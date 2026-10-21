function getCsrfToken() {
  const element = document.querySelector('meta[name="csrf-token"]');
  return element ? element.content : '';
}
