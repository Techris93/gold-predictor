self.ALLOWED_ALERT_TITLES = new Set([
  "XAUUSD Market Structure Changed",
  "XAUUSD Microstructure Changed",
  "XAUUSD Price Action Changed",
]);

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("push", (event) => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch (_) {
    data = { title: "XAUUSD Signal Changed", body: "Market signal changed" };
  }

  const title = data.title || "XAUUSD Signal Changed";
  if (!self.ALLOWED_ALERT_TITLES.has(title)) {
    return;
  }
  const options = {
    body: data.body || "Market signal changed",
    icon: "/static/favicon.png",
    badge: "/static/favicon.png",
    tag: data.tag || "xauusd-alert",
    renotify: false,
    data: {
      url: data.url || "/",
    },
  };

  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const targetUrl = (event.notification.data && event.notification.data.url) || "/";

  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((clients) => {
      for (const client of clients) {
        if (client.url.includes(targetUrl) && "focus" in client) {
          return client.focus();
        }
      }
      if (self.clients.openWindow) {
        return self.clients.openWindow(targetUrl);
      }
      return null;
    }),
  );
});
