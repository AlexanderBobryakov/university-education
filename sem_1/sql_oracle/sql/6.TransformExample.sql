-- 1. ДЕЛАЕМ СХЕМУ В МОДЕЛЕРЕ -> export to DDL
-- 2. НАКАТЫВАЕМ DDL НА БД
-- 3. ЗАПОЛНЯЕМ ИЗМЕРЕНИЯ
-- пример для DIM_COUNTRY - DIM_CITY (с иерархией от самой дальней к центру)
create sequence SEQ_DIM_COUNTRY; -- для id в измерении (лучше один seq на каждую таблицу)
insert into DIM_COUNTRY(COUNTRY_ID, COUNTRY_NAME)
select SEQ_DIM_COUNTRY.nextval, country from (select distinct COUNTRY from CITY);
commit;
insert into DIM_CITY(CITY_ID, CITY_NAME, DIM_COUNTRY_COUNTRY_ID)
select CITY_ID, CITY_NAME, (select COUNTRY_ID from DIM_COUNTRY where COUNTRY_NAME = CITY.COUNTRY) from CITY;
commit;

-- генерим данные для измерения с датой (иерархия по LEVEL)
create sequence SEQ_DIM_DATE;
select min(SALES_ORDER.ORDER_DATE), max(SALES_ORDER.ORDER_DATE) from SALES_ORDER; -- смотрим разброс дат
create or replace procedure P_GENERATE_DIM_DATE
(
    DATE_FROM IN DATE,
    DATE_TO IN DATE
) AS
d_date DATE := DATE_FROM;
BEGIN
    LOOP
        -- заполняем уровен 1 (дни)
        INSERT INTO DIM_DATE(DATE_ID, DATE_NAME, "LEVEL")   -- "LEVEL" - т.к. LEVEL - ключевое слово
        values (SEQ_DIM_DATE.nextval, TO_CHAR(d_date, 'DD.MM.YYYY'), 1);
        -- заполняем уровен 2 (месяца)
        if (EXTRACT(DAY from d_date) = 1) then
            INSERT INTO DIM_DATE(DATE_ID, DATE_NAME, "LEVEL")
            values (SEQ_DIM_DATE.nextval, TO_CHAR(d_date, 'MM.YYYY'), 2);
        end if;
        -- заполняем уровен 3 (года)
        if (EXTRACT(DAY from d_date) = 1) then
            INSERT INTO DIM_DATE(DATE_ID, DATE_NAME, "LEVEL")
            values (SEQ_DIM_DATE.nextval, TO_CHAR(d_date, 'YYYY'), 3);
        end if;
        d_date := d_date + 1;

        EXIT WHEN d_date >= DATE_TO;
    end loop;
END;
-- запустим ее
begin
    P_GENERATE_DIM_DATE(TO_DATE('01.01.2000', 'DD.MM.YYYY'), TO_DATE('01.01.2017', 'DD.MM.YYYY'));
end;

-- 4. ЗАПОЛНЯЕМ ТАБЛИЦУ ФАКТОВ
-- грузим данные без трансформации
create table STG_FACT_SALES
as select ol.ORDER_LINE_ID,
          ol.product_id,
          so.SALES_ORDER_ID,
          so.manager_id,
          m.MANAGER_FIRST_NAME,
          m.MANAGER_LAST_NAME,
          m.office_id,
          o.OFFICE_NAME,
          o.city_id,
          C2.CITY_NAME,
          C2.COUNTRY,
          C2.REGION,
          ol.product_qty,
          ol.product_price,
          ol.product_qty * ol.product_price as sale_amount,
          so.order_date
    from SALES_ORDER_LINE ol
    inner join SALES_ORDER SO on SALES_ORDER_LINE.SALES_ORDER_ID = SO.SALES_ORDER_ID
    inner join MANAGER M on SO.MANAGER_ID = M.MANAGER_ID
    inner join OFFICE O on M.OFFICE_ID = O.OFFICE_ID
    inner join CITY C2 on O.CITY_ID = C2.CITY_ID;
-- чистим при необходимости от дубликатов и тд
-- заполняем таблицу фактов
insert into FACT_SALES(...)
select
    (...)
    (select DATE_ID from DIM_DATE D where D_DATE_NAME = TO_CHAR(STG_FACT_SALES.order_date, 'DD.MM.YYYY'))
from STG_FACT_SALES;
commit;

-- 5. РАСЧЕТ АГРЕГАТОВ -предрассчитанные суммы по группировки измерений
select PRODUCT_ID, NULL, NULL,  SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY) -- все измерения должны сответсвовать FACT_SALES
                                                                                -- записываем NULL
from FACT_SALES
group by PRODUCT_ID
    union all
select PRODUCT_ID, MANAGER_ID, NULL, NULL, SUM()...
    from FACT_SALES
group by PRODUCT_ID, MANAGER_IF
    union all
select  NULL, NULL, SUM()...
    from FACT_SALES


-- можно заменить более вростым запросом
select  PRODUCT_ID, NULL, NULL,  SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY)
from FACT_SALES
group by CUBE(PRODUCT_ID, MANAGER_IF)--сгруппирует по PRODUCT_ID потом по MANAGER_IF потом по обоим потом без группирровки
-- и все объединит
-- если добавить все измерения в CUBE -> получим все  возможные разбиения группировок

-- для иерархических измерений (все сочетания CUBE помноженнные на все иерархические сочетания rollup)
select PRODUCT_ID, NULL, NULL,  SUM(SALE_AMOUNT), AVG(SALE_PRICE), SUM(SALE_QTY)
from FACT_SALES
group by CUBE(PRODUCT_ID, MANAGER_IF), rollup (COUNTRY_ID, CITY_ID, OFFICE_ID)  -- !!! от более крупного к мелкому

-- в DATE_ID нужно агрегировать по уровню - проблема

-- последний select можно положить в FACT_SALES_AVG (доп таблица фактов) либо в исходную FACT_SALES
create FACT_SALES_AVG as select * from FACT_SALES where ROWNUM < 1;
insert into FACT_SALES_AVG
select (...) -- пред. запрос
from FACT_SALES
group by CUBE(PRODUCT_ID, MANAGER_IF, DATE_ID), ROLLUP (COUNTRY_ID, CITY_ID, OFFICE_ID);

-- ДЛЯ ОБНОВЛЕНИЯ ДАННЫХ МОЖНО ИСПОЛЬЗОВАТЬ МАТЕРИАЛИЗОВАННОЕ ПРЕДСТАВЛЕНИЕ
create materialized view MV_FACT_SALES_AVG
as select (...) from FACT_SALES group by CUBE(...), rollup (...)