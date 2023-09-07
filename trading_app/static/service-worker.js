self.addEventListener('push', function(event) {
    const options = {
        body: event.data.text(),
        icon: 'notification_icon.png',
        badge: 'notification_icon.png',
    };

    event.waitUntil(
        self.registration.showNotification('Trading notifications', options)
    );
});

