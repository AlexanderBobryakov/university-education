-- 1.Выбрать все заказы
select *
from SALES_ORDER;

-- 2.Выбрать все заказы, введенные после 1янв2016
select *
from SALES_ORDER
WHERE order_date > to_date('01-01-2016', 'DD-MM-YYYY');

-- 3.Выбрать все заказы, введенные после 1янв2016 и до 15 июля 2016 года
select *
from SALES_ORDER
WHERE order_date > to_date('01-01-2016', 'DD-MM-YYYY')
  and order_date < to_date('15-07-2016', 'DD-MM-YYYY');

-- 4.Найти менеджеров с именем 'Henry'
select *
from MANAGER
WHERE manager_first_name = 'Henry';

-- 5.Выбрать все заказы менеджеров с именем Henry
select *
from SALES_ORDER O
         INNER JOIN MANAGER M
                    ON O.MANAGER_ID = M.MANAGER_ID
WHERE manager_first_name = 'Henry';

-- 6.Выбрать все уникальные страны из таблицы CITY
select DISTINCT *
FROM CITY;

-- 7.Выбрать все уникальные комбинации страны и региона из таблицы CITY
select DISTINCT REGION, COUNTRY
FROM CITY
ORDER BY COUNTRY;

-- 8.Выбрать все страны из таблицы CITY с количеством городов в них.
select COUNTRY, COUNT(CITY_ID)
FROM CITY
GROUP BY COUNTRY;

-- 9.Выбрать количество товаров (QTY), проданное с 1 по 30 января 2016 года
select sum(PRODUCT_QTY)
from SALES_ORDER_LINE sol
         left outer join SALES_ORDER so on sol.SALES_ORDER_ID = so.SALES_ORDER_ID
where so.ORDER_DATE between '01.01.2016' and '30.01.2016';

-- 10.Выбрать все уникальные названия городов, регионов и стран в одной колонке
select distinct CITY_NAME
FROM CITY
UNION ALL 
select distinct REGION
FROM CITY
UNION ALL 
select distinct COUNTRY
FROM CITY;

-- 11.Вывести имена и фамилии менеджер(ов), продавшего товаров в январе 2016 года на наибольшую сумму.
with tempTable as (
    --  все менеджеры с их суммой продажи
    SELECT MANAGER_ID,
           sum(PRODUCT_PRICE * PRODUCT_QTY) TOTAL
    FROM SALES_ORDER_LINE L
             INNER JOIN SALES_ORDER S
                        ON L.SALES_ORDER_ID = S.SALES_ORDER_ID
    WHERE order_date >= to_date('01-01-2016', 'DD-MM-YYYY')
      and order_date <= to_date('31-01-2016', 'DD-MM-YYYY')
    GROUP BY MANAGER_ID
)
SELECT MANAGER_FIRST_NAME, MANAGER_LAST_NAME
FROM MANAGER
WHERE MANAGER_ID IN (
    SELECT MANAGER_ID
    FROM tempTable
    WHERE TOTAL = (
        SELECT MAX(TOTAL)
        FROM tempTable
    )
);