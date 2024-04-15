from django.conf import settings
from django.db import connection
from django.utils.deprecation import MiddlewareMixin
import time
import logging

logger = logging.getLogger('db_query_logger')

class QueryLogMiddleware(MiddlewareMixin):
    def process_request(self, request):
        if settings.DB_QUERY_LOGGING_ENABLED:
            request._query_log_start_time = time.time()

    def process_response(self, request, response):
        if not settings.DB_QUERY_LOGGING_ENABLED:
            return response

        total_time = time.time() - request._query_log_start_time
        total_queries = len(connection.queries)
        total_query_time = sum(float(query['time']) for query in connection.queries)

        if hasattr(request, 'resolver_match'):
            if request.resolver_match is not None:
                view_name = '"{}"'.format(request.resolver_match.view_name)
            else:
                view_name = '"Unknown"'
        else:
            view_name = '"Unknown"'

        logger.info(f"View: {view_name}, Total Queries: {total_queries}, Total Query Time: {total_query_time}s, Total Request Time: {total_time}s")

        return response
