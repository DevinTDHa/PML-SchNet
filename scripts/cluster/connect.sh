#!/bin/bash
chmod 0400 ssh_keys/pml_id_rsa.pub
ssh -F ssh_config cluster
