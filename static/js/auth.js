document.querySelectorAll('[data-password-toggle]').forEach((button) => {
  button.addEventListener('click', () => {
    const input = document.querySelector(button.dataset.passwordToggle);
    if (!input) return;
    const isPassword = input.type === 'password';
    input.type = isPassword ? 'text' : 'password';
    const icon = button.querySelector('i');
    if (icon) {
      icon.className = isPassword ? 'bi bi-eye' : 'bi bi-eye-slash';
    }
  });
});
