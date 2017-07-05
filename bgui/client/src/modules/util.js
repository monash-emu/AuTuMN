
export default {
  jstr(o) {
    return JSON.stringify(o, null, 2)
  },
  removeItem(aList, item, key) {
    for (let i = aList.length - 1; i >= 0; i -= 1) {
      if (aList[i][key] === item[key]) {
        aList.splice(i, 1)
      }
    }
  },
  downloadObject (fname, obj) {
    let s = JSON.stringify(obj, null, 2)
    let data = 'text/json;charset=utf-8,' + encodeURIComponent(s);

    let a = document.createElement('a');
    a.href = 'data:' + data;
    a.download = 'data.json';
    a.innerHTML = 'download JSON';

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }
}