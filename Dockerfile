FROM ghcr.io/prefix-dev/pixi:0.39.4 AS build

# copy source code, pixi.toml and pixi.lock to the container
WORKDIR /app
COPY . .
RUN pixi install --frozen -e dev
# create the shell-hook bash script to activate the environment
RUN pixi shell-hook --frozen -e dev -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh
RUN cat /shell-hook >> /app/entrypoint.sh
# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /app/entrypoint.sh

FROM ubuntu:24.04 AS production
WORKDIR /app
COPY --from=build /app /app
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh

ENTRYPOINT [ "/app/entrypoint.sh" ]
CMD [ "python3", "/app/examples/swarm-monitoring/run_hscc_experiments.py" ]

