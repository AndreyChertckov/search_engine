FROM nginx

COPY ./nginx.conf /etc/nginx/conf.d/default.conf
COPY ./index.html /var/www/

CMD ["nginx" , "-g", "daemon off;"]
