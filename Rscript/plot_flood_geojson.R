library(dplyr) 
library(leaflet)
library(mapdata)
library(ggplot2)
library(ggmap)
library(geojsonio)
library(maptools)

region <- "../data/region.json"
shapefile <- "../data/dki_kelurahan.shp"
flood <- "../data/kejadian_banjir.csv" #confidential data
monthly_flood <- "../data/banjir_bulanan.csv" #confidential data

topoData <- readLines(region) %>% paste(collapse = "\n")

#coba plot using leaflet
leaflet() %>% setView(lng = 106.8227, lat = 6.1745, zoom = 1) %>%
  addTiles() %>%
  addGeoJSON(topoData, weight = 1, color = "#444444", fill = FALSE)

#plot from shape
map_jakarta <- readShapeSpatial(shapefile)
d <- data.frame(Id = map_jakarta$ID, kecamatan = map_jakarta$Kecamatan, 
                kelurahan = map_jakarta$KEL_NAME)
gpclibPermit()

ggplot(map_jakarta) +
  aes(long,lat,group=group, fill = FALSE) + 
  geom_polygon() + 
  theme(legend.position="none") +
  geom_path(color="black") +
  coord_equal() +
  scale_fill_brewer("Bydeler")

#proccess flood data
banjir <- read.csv(flood)
banjir$Ketinggian.Air.Min = as.numeric(as.character(banjir$Ketinggian.Air.Min))
banjir$Ketinggian.Air.Max = as.numeric(as.character(banjir$Ketinggian.Air.Max))
banjir$Ketinggian.Average = as.numeric(as.character(banjir$Ketinggian.Average))

banjir %>% select(Tahun, Bulan, Wilayah.Administrasi, Kecamatan, Kelurahan,
                  Ketinggian.Average) %>% 
          group_by(Tahun, Bulan, Wilayah.Administrasi, Kecamatan, Kelurahan) %>%
          summarise(Ketinggian.Average = mean(Ketinggian.Average)) -> summ.banjir

#load clean data to merge with jakarta map
f_jakarta <- fortify(map_jakarta, region = "ID")
write.csv(f_jakarta, "f_jakarta.csv")
summ.banjir <- read.csv(monthly_flood)
summ.banjir$id <- as.character(summ.banjir$id)
d.2013 <- summ.banjir[summ.banjir$Tahun == 2013,]
d.2014 <- summ.banjir[summ.banjir$Tahun == 2014,]
d.2015 <- summ.banjir[summ.banjir$Tahun == 2015,]
d.2016 <- summ.banjir[summ.banjir$Tahun == 2016,]

#merge manual
map_and_data <- merge(f_jakarta, d.2013[summ.banjir$Bulan == 1,], by = 'id')
#map_and_data <- read.csv("f_jakarta2.csv")
for (bulan in 1:8)
{
  
  map_and_data <- left_join(f_jakarta, d.2016[d.2016$Bulan == bulan,], by="id")
  map_and_data$Ketinggian.Average[is.na(map_and_data$Ketinggian.Average)] = 0
  
  plot_banjir <-  ggplot(data = map_and_data, aes(x = long, y = lat, group = id))
  plot_banjir + geom_polygon(aes(fill = Ketinggian.Average)) + 
    geom_path(color="black") +
    scale_fill_gradientn("Ketinggian Air(cm)",
                         colours=(brewer.pal(8,"Blues")),
                         limits=c(0, 250)) +
    labs(x = "Longitude", y = "Latitude", fill = "Ketinggian Air(cm)", 
         title =  paste("Banjir di Jakarta ", "2016-", bulan, sep = ""))
  
  ggsave(paste("2016-", bulan, ".png", sep = ""), width = 20, height = 20, units = "cm")
  
}


#+
#labs(x = "Longitude", y = "Latitude", fill = "Ketinggian Air(cm)", 
#     title =  "Banjir di Jakarta-February 2013")