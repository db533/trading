from django.shortcuts import render
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from webpush import send_user_notification

@csrf_exempt
#@login_required  # You can require authentication here if needed
def subscription_endpoint(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            subscription = data.get('subscription')

            # Store the subscription information in your database for later use
            # You can also associate subscriptions with specific users

            # Trigger a push notification
            send_user_notification(
                user=request.user,  # Replace with your user object
                payload={
                    'head': 'New Notification',
                    'body': 'This is a sample notification.',
                    'icon': '/static/notification_icon.png',  # Replace with your icon path
                },
                ttl=1000,  # Time to live in seconds
            )

            return JsonResponse({'message': 'Subscription data saved successfully'})
        except json.JSONDecodeError as e:
            print(f'JSON decoding error: {str(e)}')
            return JsonResponse({'message': 'Invalid JSON format'}, status=400)
    else:
        return JsonResponse({'message': 'Invalid request method'}, status=405)

def subscription_page(request):
    return render(request, 'notifications/notification_subscription.html')