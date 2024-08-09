// Redirect to google.com if user is found browsing facebook.com and subpages
if (window.location.href.includes('sammyboy.com')) {
  window.location.href = 'http://localhost:3000/danger';
}