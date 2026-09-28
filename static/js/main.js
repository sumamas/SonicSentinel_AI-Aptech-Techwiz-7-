const navToggle = document.getElementById('navToggle');
const mainNav = document.getElementById('mainNav');
if (navToggle && mainNav) {
  navToggle.addEventListener('click', () => mainNav.classList.toggle('open'));
}

document.querySelectorAll('.nav-dropdown-toggle').forEach((toggle) => {
  toggle.addEventListener('click', (event) => {
    event.stopPropagation();
    const parent = toggle.closest('.nav-dropdown');
    document.querySelectorAll('.nav-dropdown.open').forEach((el) => { if (el !== parent) el.classList.remove('open'); });
    parent.classList.toggle('open');
  });
});

const profileMenu = document.getElementById('profileMenu');
const profileButton = document.getElementById('profileButton');
if (profileMenu && profileButton) {
  profileButton.addEventListener('click', (event) => {
    event.stopPropagation();
    const open = profileMenu.classList.toggle('open');
    profileButton.setAttribute('aria-expanded', open ? 'true' : 'false');
  });
}

document.addEventListener('click', () => {
  document.querySelectorAll('.nav-dropdown.open').forEach((el) => el.classList.remove('open'));
  if (profileMenu) profileMenu.classList.remove('open');
  if (profileButton) profileButton.setAttribute('aria-expanded', 'false');
});
