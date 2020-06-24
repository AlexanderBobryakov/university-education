-- 1. CREATE TABLES
CREATE TABLE ALEX.dim_date
(
    date_id   INTEGER NOT NULL,
    date_name VARCHAR2(100 CHAR),
    "LEVEL"   INTEGER
);
ALTER TABLE ALEX.dim_date
    ADD CONSTRAINT dim_date_pk PRIMARY KEY (date_id);

CREATE TABLE ALEX.dim_posts
(
    post_id   INTEGER NOT NULL,
    post_name VARCHAR2(1000)
);
ALTER TABLE ALEX.dim_posts
    ADD CONSTRAINT posts_pk PRIMARY KEY (post_id);

CREATE TABLE ALEX.dim_tags
(
    tag_id   INTEGER NOT NULL,
    tag_name VARCHAR2(1000)
);
ALTER TABLE ALEX.dim_tags
    ADD CONSTRAINT dim_tags_pk PRIMARY KEY (tag_id);

CREATE TABLE ALEX.fact_view
(
    fact_view_pk      INTEGER NOT NULL,
    view_count        INTEGER,
    dim_posts_post_id INTEGER NOT NULL,
    dim_date_date_id  INTEGER NOT NULL,
    dim_tags_tag_id   INTEGER NOT NULL
);
ALTER TABLE ALEX.fact_view
    ADD CONSTRAINT fact_view_pk PRIMARY KEY (fact_view_pk);
ALTER TABLE ALEX.fact_view
    ADD CONSTRAINT fact_view_dim_date_fk FOREIGN KEY (dim_date_date_id)
        REFERENCES dim_date (date_id);
ALTER TABLE ALEX.fact_view
    ADD CONSTRAINT fact_view_dim_posts_fk FOREIGN KEY (dim_posts_post_id)
        REFERENCES dim_posts (post_id);
ALTER TABLE ALEX.fact_view
    ADD CONSTRAINT fact_view_dim_tags_fk FOREIGN KEY (dim_tags_tag_id)
        REFERENCES dim_tags (tag_id);

-- 2. НАПОЛНЯЕМ ИЗМЕРЕНИЯ
insert into ALEX.DIM_POSTS (POST_ID, POST_NAME)
select distinct p.id, p.title
from SYSTEM.POSTS p
WHERE p.POSTTYPEID = 1;

insert into ALEX.DIM_TAGS (TAG_ID, TAG_NAME)
select *
from (
     select distinct t.id,
                 -- t.tagname, -- можно посмотреть корректность для console/consoleapplication -> консоль
                     nvl((select distinct st.targettagname
                          from SYSTEM.TAGSYNONYMS st
                          where t.tagname = st.sourcetagname
                             or t.tagname = st.targettagname), t.tagname)
     from SYSTEM.TAGS t);

-- дата
create sequence SEQ_DIM_DATE;
    -- смотрим разброс дат (10.10.2010 - 02.09.2018)
select min(SYSTEM.POSTS.CREATIONDATE), max(SYSTEM.POSTS.CREATIONDATE) from SYSTEM.POSTS;
create or replace procedure ALEX.P_GENERATE_DIM_DATE
(
    DATE_FROM IN DATE,
    DATE_TO IN DATE
) AS
    d_date DATE := DATE_FROM;
BEGIN
    LOOP
        -- заполняем уровен 1 (дни)
        INSERT INTO ALEX.DIM_DATE(DATE_ID, DATE_NAME, "LEVEL")
        values (SEQ_DIM_DATE.nextval, TO_CHAR(d_date, 'DD.MM.YYYY'), 1);
        -- заполняем уровен 2 (месяца)
        if (EXTRACT(DAY from d_date) = 1) then
            INSERT INTO ALEX.DIM_DATE(DATE_ID, DATE_NAME, "LEVEL")
            values (SEQ_DIM_DATE.nextval, TO_CHAR(d_date, 'MM.YYYY'), 2);
        end if;
        -- заполняем уровен 3 (года)
        if (EXTRACT(MONTH from d_date) = 1) then
            INSERT INTO ALEX.DIM_DATE(DATE_ID, DATE_NAME, "LEVEL")
            values (SEQ_DIM_DATE.nextval, TO_CHAR(d_date, 'YYYY'), 3);
        end if;
        d_date := d_date + 1;

        EXIT WHEN d_date >= DATE_TO;
    end loop;
END;
-- запустим ее
begin
    ALEX.P_GENERATE_DIM_DATE(TO_DATE('10.10.2010', 'DD.MM.YYYY'), TO_DATE('03.09.2018', 'DD.MM.YYYY'));
end;
commit;
-- 3. ВСПОМОГАТЕЛЬНАЯ ТАБЛИЦА ФАКТОВ
create table STG_FACT
as
select p.id,
       p.creationdate,
       p.viewcount,
       p.tags,
       pt.tagid,
       t.tag_name tagname
from SYSTEM.POSTS p
         inner join SYSTEM.POSTTAGS pt on p.id = pt.postid
         inner join ALEX.dim_tags t on pt.tagid = t.tag_id
order by p.id;
-- заполняем таблицу фактов
create sequence SEQ_FACT_VIEW;
insert into ALEX.FACT_VIEW (FACT_VIEW_PK,
                            VIEW_COUNT,
                            DIM_POSTS_POST_ID,
                            DIM_DATE_DATE_ID,
                            DIM_TAGS_TAG_ID)
select SEQ_FACT_VIEW.nextval,
       stg.viewcount,
       stg.id,
       (select DATE_ID from DIM_DATE where date_name = TO_CHAR(stg.creationdate, 'DD.MM.YYYY')),
       stg.tagid
from STG_FACT stg;
commit;
-- 4. АГРЕГАТЫ
CREATE TABLE ALEX.FACT_VIEW_AVG
(
    view_count        INTEGER,
    dim_posts_post_id INTEGER,
    dim_date_date_id  INTEGER,
    dim_tags_tag_id   INTEGER
);
insert into FACT_VIEW_AVG
select sum(view_count),
     dim_posts_post_id,
     dim_date_date_id,
     dim_tags_tag_id
from FACT_VIEW
group by CUBE (dim_posts_post_id, dim_date_date_id, dim_tags_tag_id);
select * from FACT_VIEW_AVG;
select * from DIM_DATE;








CREATE TABLE ALEX.FACT_VIEW_AVG_Test
(
    view_count        INTEGER,
    dim_posts_post_id INTEGER,
    dim_date_date_id INTEGER,
    dim_date_date_level  INTEGER,
    dim_tags_tag_id   INTEGER
);
insert into FACT_VIEW_AVG_Test
select sum(view_count),
       dim_posts_post_id,
       dim_date_date_id,
       d."LEVEL",
       dim_tags_tag_id
from FACT_VIEW inner join DIM_DATE d on dim_date_date_id = d.date_id
group by CUBE (dim_posts_post_id, dim_date_date_id, d."LEVEL", dim_tags_tag_id);
commit;

select * from FACT_VIEW_AVG_Test where dim_date_date_level is not null and dim_date_date_level != 1;