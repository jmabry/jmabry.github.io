ARG RUBY_VERSION=3.3.2
FROM ruby:$RUBY_VERSION-slim

RUN apt-get update \
  && apt-get install -y \
    build-essential \
    git \
    locales \
    nodejs

WORKDIR /src/site

COPY Gemfile /src/site/Gemfile

RUN NOKOGIRI_USE_SYSTEM_LIBRARIES=true bundle install

RUN \
  echo "en_US UTF-8" > /etc/locale.gen && \
  locale-gen en-US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

CMD ["jekyll", "serve", "--watch", "-H", "0.0.0.0", "-P", "4000"]