document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('formEditEmployee').addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      const response = await fetch(location.pathname, {
        method: 'POST',
        body: new FormData(event.target),
        headers: { 'X-CSRF-Token': getCsrfToken() }
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.message || 'Cập nhật thất bại');
      }
      window.location.href = '/admin/employees';
    } catch (error) {
      alert(error.message || 'Không thể kết nối đến máy chủ');
    }
  });
});
