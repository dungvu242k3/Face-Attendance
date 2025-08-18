async function deleteEmployee(id) {
  if (!confirm("Bạn có chắc chắn muốn xoá nhân viên này?")) return;
  const res = await fetch(`/admin/delete_employee/${id}`, {
    method: "POST"
  });
  const data = await res.json();
  alert(data.message || "Xong");
  if (data.status === "ok") location.reload();
}
