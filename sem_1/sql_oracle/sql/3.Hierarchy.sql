-- 1. Вывыести все директории в виде ID, Навание, Путь до корня
select id,
       name,
       SYS_CONNECT_BY_PATH(name, '/')
from FILE_SYSTEM
connect by nocycle prior id = PARENT_ID
start with id = 0;

-- 2. Для каждой директории посчитать объем занимаемого места на диске (с учетом всех сложенных папок)
select id,
       name,
       SYS_CONNECT_BY_PATH(name, '/'),
       FILE_SIZE,
       (select sum(FILE_SIZE)
        from FILE_SYSTEM inner
        connect by PARENT_ID = prior id
        start with inner.id = outer.id) as volume
from FILE_SYSTEM outer
connect by prior id = PARENT_ID
start with PARENT_ID is NULL;

-- 3. Добавить в запрос: сколько процентов директория занимает места относительно всех своих соседей
select ID,
       name,
       path,
       type,
       FILE_SIZE,
       PARENT_ID,
       volume,
       RATIO_TO_REPORT(volume) OVER (PARTITION BY PARENT_ID)
from (
         select ID,
                PARENT_ID,
                type,
                name,
                SYS_CONNECT_BY_PATH(name, '/')   as path,
                FILE_SIZE,
                (select sum(FILE_SIZE)
                 from FILE_SYSTEM inner
                 connect by PARENT_ID = prior id
                 start with inner.id = outer.id) as volume
         from FILE_SYSTEM outer
         connect by prior id = PARENT_ID
         start with PARENT_ID is NULL
     )
order by PARENT_ID;