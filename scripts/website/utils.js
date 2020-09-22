export const fileToDate = ({ filename }) => {
  let dateStr = filename.split('.')[0].split('-').slice(3, 8).join('-')
  dateStr =
    dateStr.split('T')[0] + 'T' + dateStr.split('T')[1].split('-').join(':')
  return Date.parse(dateStr) / 1000
}
