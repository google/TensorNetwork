FROM ubuntu

# Install basic tools.
RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
     python3-pip python3-tk git emacs vim locales

# Configure UTF-8 encoding.
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Make python3 default
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

# Install Tensor Network with the needed Python libraries.
RUN pip3 install tensornetwork

# Clone the repository so development work can be done in the container.
RUN git clone https://github.com/google/TensorNetwork

WORKDIR /TensorNetwork/examples

EXPOSE 8888
