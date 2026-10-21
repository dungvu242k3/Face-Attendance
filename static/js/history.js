function filterHistoryTable() {
  const filterType = document.getElementById('filterType').value.toLowerCase();
  const searchName = document.getElementById('searchName').value.toLowerCase();
  document.querySelectorAll('#historyTable tbody tr').forEach((row) => {
    const name = row.cells[2].textContent.toLowerCase();
    const type = row.cells[5].textContent.toLowerCase();
    row.style.display = (!filterType || type === filterType) && (!searchName || name.includes(searchName)) ? '' : 'none';
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const filterType = document.getElementById('filterType');
  const searchName = document.getElementById('searchName');
  if (!filterType || !searchName) return;
  filterType.addEventListener('change', filterHistoryTable);
  searchName.addEventListener('input', filterHistoryTable);
});
