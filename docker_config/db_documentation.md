# Database Documentation
We will be using mongodb with docker to provide environment encapsulation
so that the db can be deployed in any environment.

you will need docker pre-installed on your computer before u can proceed.

### youtube guides
- [Learn how to use postgres sql with docker](https://www.youtube.com/watch?v=A8dErdDMqb0)
- [Docker with persistent storage](https://www.youtube.com/watch?v=G3gnMSyX-XM&t=1s)
- [Docker Volumes Setup](https://www.youtube.com/watch?v=G-5c25DYnfI)

**We will be using postgres database for this project.**

## Programs to download
Containerisation technology:
- Docker

Choose either one (Database viewer):
- Dbeaver
- Pgadmin

## MongoDB version used
- version used: 12-alpine

command to pull from docker repository
```bash
docker pull postgres:12-alpine
```

## Getting Started (the TLDR version of it)

How to get started using the database

1. ensure docker is installed on your computer
2. open a terminal in the current directory
3. run this command in this current directory of this file

```bash
docker-compose up
```

Thats it, 3 steps to get started using this postgres

## shutdown the db_server
```bash
docker-compose down
```

## Database Credentials
| Syntax      | Info to Enter |
| ----------- | ------------- |
| Username    | test          |
| Password    | secret        |
| Database    | demo          |
| Host        |localhost      |

## for viewing of DB using GUI (optional)

<img width="600" alt="Screenshot 2021-10-04 at 3 56 18 PM" src="https://user-images.githubusercontent.com/22993048/135814703-e9031003-a9f2-4520-bb45-08add39538fb.png">
