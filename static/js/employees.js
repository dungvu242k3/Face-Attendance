document.addEventListener('DOMContentLoaded', () => {
  const nameInput = document.getElementById('searchName');
  const positionSelect = document.getElementById('filterPosition');

  function filterEmployees() {
    const name = nameInput.value.toLowerCase();
    const position = positionSelect.value;
    document.querySelectorAll('tbody tr').forEach((row) => {
      const rowName = row.children[1].textContent.toLowerCase();
      const rowPosition = row.children[4].textContent;
      row.style.display = rowName.includes(name) && (!position || rowPosition === position) ? '' : 'none';
    });
  }

  nameInput.addEventListener('input', filterEmployees);
  positionSelect.addEventListener('change', filterEmployees);

  document.querySelectorAll('.delete-btn').forEach((button) => {
    button.addEventListener('click', async () => {
      if (!confirm('Bạn có chắc muốn xóa nhân viên này?')) return;
      const empId = button.dataset.empId;
      try {
        const response = await fetch(`/admin/delete_employee/${empId}`, {
          method: 'POST',
          headers: { 'X-CSRF-Token': getCsrfToken() }
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
          throw new Error(data.message || 'Xóa thất bại');
        }
        button.closest('tr').remove();
        filterEmployees();
      } catch (error) {
        alert(error.message || 'Không thể kết nối đến máy chủ');
      }
    });
  });
});
